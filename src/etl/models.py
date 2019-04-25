from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy import Column, Integer, Boolean, Date, Float, \
    CHAR, BIGINT, VARCHAR, TIMESTAMP

Base = declarative_base()


class User(Base):
    __tablename__ = 'users'
    id = Column(UUID(as_uuid=True),
                primary_key=True)
    has_email = Column(Boolean,
                       nullable=False)
    phone_country = Column(VARCHAR(300))
    is_fraudster = Column(Boolean,
                          nullable=False,
                          default=False)
    terms_version = Column(Date)
    created_date = Column(TIMESTAMP,
                          nullable=False)
    state = Column(VARCHAR(25),
                   nullable=False)
    country = Column(VARCHAR(2))
    birth_year = Column(Integer)
    kyc = Column(VARCHAR(20))
    failed_sign_in_attempts = Column(Integer)

    def __repr__(self):
        return "<User(id='{}', is_fraud='{}', country={})>" \
            .format(self.id, self.is_fraudster, self.country)


class Transaction(Base):
    __tablename__ = 'transactions'
    id = Column(UUID(as_uuid=True),
                primary_key=True)
    currency = Column(CHAR(3),
                      nullable=False)
    amount = Column(BIGINT,
                    nullable=False)
    state = Column(VARCHAR(25),
                   nullable=False)
    created_date = Column(TIMESTAMP,
                          nullable=False)
    merchant_category = Column(VARCHAR(100))
    merchant_country = Column(VARCHAR(3))
    entry_method = Column(VARCHAR(4),
                          nullable=False)
    user_id = Column(UUID(as_uuid=True),
                     nullable=False)

    type = Column(VARCHAR(20),
                  nullable=False)
    source = Column(VARCHAR(20),
                    nullable=False)

    def __repr__(self):
        return "<Transaction(id='{}', state='{}', amount={}, user={})>" \
            .format(self.id, self.state, self.amount, self.user_id)


class FxRates(Base):
    __tablename__ = 'fx_rates'
    ts = Column(TIMESTAMP(timezone=False),
                primary_key=True)
    base_ccy = Column(VARCHAR(3),
                      primary_key=True)
    ccy = Column(VARCHAR(10),
                 primary_key=True)
    rate = Column(Float)

    def __repr__(self):
        return "<FxRates(ts='{}', base_ccy='{}', ccy={}, rate={})>" \
            .format(self.ts, self.base_ccy, self.ccy, self.rate)


class CurrencyDetails(Base):
    __tablename__ = 'currency_details'
    ccy = Column(VARCHAR(10), primary_key=True)
    iso_code = Column(Integer)
    exponent = Column(Integer)
    is_crypto = Column(Boolean, nullable=False)

    def __repr__(self):
        return "<CurrencyDetails(ccy='{}', iso_code='{}', exponent={}, is_crypto={})>" \
            .format(self.ccy, self.iso_code, self.exponent, self.is_crypto)
